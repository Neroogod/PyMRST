import { FormEvent, useEffect, useState } from "react";
import {
  api, getAccessToken, hasAccessToken, setAccessToken, type Capability, type ProjectRecord,
  type SecurityActor, type SecurityAuditEvent, type SecurityToken, type SecurityUser,
} from "../api/client";

const allScopes: Capability[] = ["system_admin", "view", "edit", "execute", "publish", "audit"];

export function SecuritySettingsPanel() {
  const [mode, setMode] = useState<"local" | "shared">("local");
  const [identity, setIdentity] = useState<SecurityActor | null>(null);
  const [users, setUsers] = useState<SecurityUser[]>([]);
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [tokens, setTokens] = useState<SecurityToken[]>([]);
  const [events, setEvents] = useState<SecurityAuditEvent[]>([]);
  const [chain, setChain] = useState<{ valid: boolean; events: number; head_hash?: string } | null>(null);
  const [selectedUser, setSelectedUser] = useState("");
  const [name, setName] = useState(""); const [email, setEmail] = useState("");
  const [role, setRole] = useState<SecurityUser["system_role"]>("engineer");
  const [tokenLabel, setTokenLabel] = useState("Studio session");
  const [expiryDays, setExpiryDays] = useState(30);
  const [issuedSecret, setIssuedSecret] = useState("");
  const [loginToken, setLoginToken] = useState("");
  const [membershipProject, setMembershipProject] = useState("");
  const [membershipRole, setMembershipRole] = useState<"owner" | "editor" | "runner" | "analyst" | "viewer">("viewer");
  const [busy, setBusy] = useState(""); const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  async function reload(signal?: AbortSignal) {
    const current = await api.securityIdentity(signal);
    setIdentity(current);
    const currentMode = await api.securityMode(signal); setMode(currentMode.mode);
    if (current.system_role === "admin" || current.local_session) {
      const [directory, projectList, audit, verification] = await Promise.all([
        api.securityUsers(signal), api.projects(signal), api.securityAudit(20, signal),
        api.verifySecurityAudit(signal),
      ]);
      setUsers(directory); setProjects(projectList); setEvents(audit); setChain(verification);
      setSelectedUser((value) => value || directory[0]?.user_id || "");
      setMembershipProject((value) => value || projectList[0]?.project_id || "");
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal).catch((caught) => {
      if (caught.name !== "AbortError") setError(caught.message);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedUser) { setTokens([]); return; }
    api.securityTokens(selectedUser).then(setTokens).catch((caught) => setError(caught.message));
  }, [selectedUser]);

  async function createUser(event: FormEvent) {
    event.preventDefault(); setBusy("user"); setError("");
    try {
      const user = await api.createSecurityUser({ display_name: name, email, system_role: role });
      setUsers((value) => [...value, user]); setSelectedUser(user.user_id);
      setName(""); setEmail(""); setNotice("Identity created. Assign only the access needed for its work.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not create identity"); }
    finally { setBusy(""); }
  }

  async function issueToken() {
    if (!selectedUser) return; setBusy("token"); setError(""); setIssuedSecret("");
    const selected = users.find((user) => user.user_id === selectedUser);
    const scopes = selected?.system_role === "admin" ? allScopes : allScopes.filter((value) => value !== "system_admin");
    try {
      const expires = new Date(Date.now() + expiryDays * 86400_000).toISOString();
      const issued = await api.issueSecurityToken(selectedUser, { label: tokenLabel, scopes, expires_at: expires });
      setTokens((value) => [...value, issued]); setIssuedSecret(issued.token ?? "");
      setNotice("Token created. Copy or activate it now; the plaintext cannot be retrieved later.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not issue token"); }
    finally { setBusy(""); }
  }

  async function activateShared() {
    setBusy("mode"); setError("");
    try {
      if (!hasAccessToken()) throw new Error("Activate an administrator token in this session before enabling shared mode.");
      await api.setSecurityMode("shared"); setMode("shared");
      setNotice("Shared mode enabled. All non-health API requests now require authenticated, scoped access.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not change deployment mode"); }
    finally { setBusy(""); }
  }

  async function returnToLocal() {
    setBusy("mode"); setError("");
    try {
      await api.setSecurityMode("local"); setMode("local");
      setNotice("Local desktop mode restored. Stored users and audit history were retained.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not change deployment mode"); }
    finally { setBusy(""); }
  }

  async function activateCredential(value: string) {
    const previous = getAccessToken(); setAccessToken(value); setBusy("login"); setError("");
    try { await reload(); setIssuedSecret(""); setLoginToken(""); setNotice("Credential active for this browser tab."); }
    catch (caught) {
      setAccessToken(previous);
      setError(caught instanceof Error ? caught.message : "Credential rejected");
    } finally { setBusy(""); }
  }

  async function assignMembership() {
    if (!selectedUser || !membershipProject) return; setBusy("membership"); setError("");
    try {
      await api.setProjectMembership(membershipProject, selectedUser, membershipRole);
      setNotice("Project role assigned and recorded in the security audit chain.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not assign role"); }
    finally { setBusy(""); }
  }

  return <section className="security-settings">
    <div className="settings-intro"><span className="eyebrow">Deployment trust boundary</span><h2>Identity and access</h2><p>Local mode preserves the single-engineer desktop workflow. Shared mode requires hashed personal access tokens, narrows every request by global and project role, and records decisions in a tamper-evident audit chain.</p></div>
    <div className={`security-mode ${mode}`}><div><strong>{mode === "local" ? "Local desktop mode" : "Shared authenticated mode"}</strong><small>{identity ? `${identity.display_name} · ${identity.system_role}` : "No authenticated identity"}</small></div><b>{mode}</b>{mode === "local" ? <button disabled={busy === "mode"} onClick={activateShared}>Enable shared mode</button> : <button disabled={busy === "mode"} onClick={returnToLocal}>Return to local mode</button>}</div>
    <div className="security-login"><input id="security-login-token" name="security-login-token" type="password" autoComplete="off" placeholder="Paste a PyMRST access token for this tab" value={loginToken} onChange={(event) => setLoginToken(event.target.value)} /><button disabled={!loginToken.trim() || busy === "login"} onClick={() => activateCredential(loginToken)}>Authenticate session</button></div>
    <div className="security-grid">
      <form onSubmit={createUser}><span className="eyebrow">Directory</span><h3>Create governed identity</h3><label htmlFor="security-user-name">Display name<input id="security-user-name" name="security-user-name" required value={name} onChange={(event) => setName(event.target.value)} /></label><label htmlFor="security-user-email">Email<input id="security-user-email" name="security-user-email" required type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label><label htmlFor="security-user-role">System role<select id="security-user-role" name="security-user-role" value={role} onChange={(event) => setRole(event.target.value as SecurityUser["system_role"])}><option value="engineer">Engineer</option><option value="viewer">Viewer</option><option value="auditor">Auditor</option><option value="admin">Administrator</option></select></label><button disabled={busy === "user"}>Create identity</button></form>
      <article><span className="eyebrow">Least privilege</span><h3>Users and project roles</h3><select id="security-selected-user" name="security-selected-user" aria-label="Selected user" className="wide" value={selectedUser} onChange={(event) => setSelectedUser(event.target.value)}>{users.map((user) => <option key={user.user_id} value={user.user_id}>{user.display_name} · {user.system_role}{user.active ? "" : " · disabled"}</option>)}</select><div className="membership-row"><select id="security-membership-project" name="security-membership-project" aria-label="Membership project" value={membershipProject} onChange={(event) => setMembershipProject(event.target.value)}>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select><select id="security-membership-role" name="security-membership-role" aria-label="Membership role" value={membershipRole} onChange={(event) => setMembershipRole(event.target.value as typeof membershipRole)}><option value="viewer">Viewer</option><option value="analyst">Analyst</option><option value="runner">Runner</option><option value="editor">Editor</option><option value="owner">Owner</option></select><button disabled={!selectedUser || !membershipProject || busy === "membership"} onClick={assignMembership}>Assign</button></div><div className="token-issuer"><label htmlFor="security-token-label">Token label<input id="security-token-label" name="security-token-label" value={tokenLabel} onChange={(event) => setTokenLabel(event.target.value)} /></label><label htmlFor="security-token-expiry">Expires in days<input id="security-token-expiry" name="security-token-expiry" type="number" min="1" max="365" value={expiryDays} onChange={(event) => setExpiryDays(Number(event.target.value))} /></label><button disabled={!selectedUser || busy === "token"} onClick={issueToken}>Issue scoped token</button></div>{tokens.map((token) => <div className="token-row" key={token.token_id}><code>{token.token_prefix}…</code><span>{token.revoked_at ? "revoked" : token.expires_at ? `expires ${new Date(token.expires_at).toLocaleDateString()}` : "no expiry"}</span></div>)}</article>
      <article><span className="eyebrow">Tamper evidence</span><h3>Security audit</h3><div className={`chain-state ${chain?.valid ? "valid" : "invalid"}`}><strong>{chain?.valid ? "Chain verified" : "Chain not verified"}</strong><small>{chain?.events ?? 0} events · {chain?.head_hash?.slice(0, 14) ?? "no head"}…</small></div><div className="audit-list">{events.map((event) => <div key={event.event_id}><span className={event.outcome}>{event.outcome}</span><strong>{event.action}</strong><small>{event.actor_id} · {new Date(event.created_at).toLocaleString()}</small></div>)}</div></article>
    </div>
    {issuedSecret && <div className="issued-secret"><strong>Copy this token now—it is shown once.</strong><code>{issuedSecret}</code><button onClick={() => activateCredential(issuedSecret)}>Use in this session</button></div>}
    {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
  </section>;
}
