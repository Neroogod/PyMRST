import { FormEvent, useEffect, useState } from "react";
import {
  api, type ReasoningProviderCheckRecord, type ReasoningProviderRecord,
  type ReasoningProviderEvaluationRecord,
} from "../api/client";
import { BackupSettingsPanel } from "./BackupSettingsPanel";
import { SecuritySettingsPanel } from "./SecuritySettingsPanel";
import { InstallationSettingsPanel } from "./InstallationSettingsPanel";
import { UpdateSettingsPanel } from "./UpdateSettingsPanel";
import { WorkloadQualificationPanel } from "./WorkloadQualificationPanel";

type Kind = "openai_responses" | "openai_compatible" | "ollama";

const defaults: Record<Kind, { endpoint: string; credential: string }> = {
  openai_responses: { endpoint: "https://api.openai.com/v1", credential: "OPENAI_API_KEY" },
  openai_compatible: { endpoint: "http://127.0.0.1:8000/v1", credential: "" },
  ollama: { endpoint: "http://127.0.0.1:11434", credential: "" },
};

export function ProviderSettingsWorkspace() {
  const [providers, setProviders] = useState<ReasoningProviderRecord[]>([]);
  const [checks, setChecks] = useState<Record<string, ReasoningProviderCheckRecord[]>>({});
  const [evaluations, setEvaluations] = useState<ReasoningProviderEvaluationRecord[]>([]);
  const [providerId, setProviderId] = useState("");
  const [kind, setKind] = useState<Kind>("ollama");
  const [model, setModel] = useState("");
  const [endpoint, setEndpoint] = useState(defaults.ollama.endpoint);
  const [credentialEnv, setCredentialEnv] = useState("");
  const [timeout, setTimeoutValue] = useState(30);
  const [maximumTokens, setMaximumTokens] = useState(2048);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  async function reload(signal?: AbortSignal) {
    const [values, evaluationValues] = await Promise.all([
      api.reasoningProviders(signal), api.reasoningProviderEvaluations(undefined, signal),
    ]);
    setProviders(values);
    setEvaluations(evaluationValues);
    const histories = await Promise.all(values.map(async (item) => [
      item.provider_id, await api.reasoningProviderChecks(item.provider_id, signal),
    ] as const));
    setChecks(Object.fromEntries(histories));
  }

  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal).catch((caught) => {
      if (caught.name !== "AbortError") setError(caught.message);
    });
    return () => controller.abort();
  }, []);

  function changeKind(value: Kind) {
    setKind(value); setEndpoint(defaults[value].endpoint);
    setCredentialEnv(defaults[value].credential);
  }

  async function register(event: FormEvent) {
    event.preventDefault(); setBusy("register"); setError(""); setNotice("");
    try {
      await api.registerReasoningProvider({
        provider_id: providerId, kind, model, endpoint,
        credential_env: credentialEnv, timeout_seconds: timeout,
        maximum_output_tokens: maximumTokens, maximum_response_bytes: 1_000_000,
        enabled: true, options: {},
      });
      await reload(); setProviderId(""); setModel("");
      setNotice("Provider registered. Credentials remain outside the project database.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Registration failed"); }
    finally { setBusy(""); }
  }

  async function toggle(item: ReasoningProviderRecord) {
    setBusy(item.provider_id); setError("");
    try {
      const updated = await api.setReasoningProviderEnabled(item.provider_id, !item.enabled);
      setProviders((current) => current.map((value) => value.provider_id === updated.provider_id ? updated : value));
      setNotice(`${updated.provider_id} ${updated.enabled ? "enabled" : "disabled"}.`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Update failed"); }
    finally { setBusy(""); }
  }

  async function test(item: ReasoningProviderRecord) {
    setBusy(`check:${item.provider_id}`); setError(""); setNotice("");
    try {
      const result = await api.checkReasoningProvider(item.provider_id);
      setChecks((current) => ({ ...current, [item.provider_id]: [...(current[item.provider_id] ?? []), result] }));
      setNotice(result.status === "passed" ? "Synthetic planning handshake passed." : "Handshake failed; inspect the retained diagnostic.");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Connectivity check failed"); }
    finally { setBusy(""); }
  }

  async function evaluate(provider: string) {
    setBusy(`evaluate:${provider}`); setError(""); setNotice("");
    try {
      const result = await api.evaluateReasoningProvider(provider);
      setEvaluations((current) => [...current, result]);
      setNotice(`${provider} evaluation ${result.status}.`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Evaluation failed"); }
    finally { setBusy(""); }
  }

  return <><section className="provider-settings">
    <div className="settings-intro"><span className="eyebrow">Reasoning boundary</span><h2>Planning providers</h2><p>Optional models may propose a bounded tool graph. PyMRST independently validates every call and retains numerical authority. Secrets are read from environment variables and are never entered or stored here.</p></div>
    <div className="provider-layout">
      <form className="provider-form" onSubmit={register}>
        <div><span className="eyebrow">New provider</span><h3>Register immutable configuration</h3></div>
        <label htmlFor="provider-id">Provider ID<input id="provider-id" name="provider-id" required value={providerId} onChange={(event) => setProviderId(event.target.value)} placeholder="local_planner" /></label>
        <label htmlFor="provider-adapter">Adapter<select id="provider-adapter" name="provider-adapter" value={kind} onChange={(event) => changeKind(event.target.value as Kind)}><option value="ollama">Ollama · local</option><option value="openai_compatible">OpenAI-compatible · local</option><option value="openai_responses">OpenAI Responses · cloud</option></select></label>
        <label htmlFor="provider-model">Model<input id="provider-model" name="provider-model" required value={model} onChange={(event) => setModel(event.target.value)} placeholder={kind === "ollama" ? "qwen3:8b" : "Configured model name"} /></label>
        <label htmlFor="provider-endpoint">Endpoint<input id="provider-endpoint" name="provider-endpoint" required value={endpoint} onChange={(event) => setEndpoint(event.target.value)} /></label>
        <label htmlFor="provider-credential-env">Credential environment variable<input id="provider-credential-env" name="provider-credential-env" value={credentialEnv} onChange={(event) => setCredentialEnv(event.target.value)} placeholder="No credential required" /><small>Only the variable name is stored.</small></label>
        <div className="provider-limits"><label htmlFor="provider-timeout">Timeout, seconds<input id="provider-timeout" name="provider-timeout" type="number" min="1" max="300" value={timeout} onChange={(event) => setTimeoutValue(Number(event.target.value))} /></label><label htmlFor="provider-max-tokens">Maximum output tokens<input id="provider-max-tokens" name="provider-max-tokens" type="number" min="128" max="16384" value={maximumTokens} onChange={(event) => setMaximumTokens(Number(event.target.value))} /></label></div>
        <button className="button primary" disabled={busy === "register"}>{busy === "register" ? "Registering…" : "Register provider"}</button>
      </form>
      <div className="provider-catalog"><div><span className="eyebrow">Governed registry</span><h3>Available planners</h3></div><article className="baseline-provider"><div className="provider-card-head"><span className="provider-health passed" /><div><strong>deterministic-v1</strong><small>Offline reference planner</small></div><b>Always available</b></div>{(() => { const latest = evaluations.filter((value) => value.provider_id === "deterministic-v1").at(-1); return <><dl><dt>Qualification</dt><dd>{latest ? `${latest.status} · ${(latest.report.summary.pass_rate * 100).toFixed(0)}% pass` : "Not evaluated"}</dd><dt>Mean F1</dt><dd>{latest ? latest.report.summary.mean_f1.toFixed(3) : "—"}</dd><dt>External data</dt><dd>None</dd></dl><div className="provider-actions"><button disabled={Boolean(busy)} onClick={() => evaluate("deterministic-v1")}>{busy === "evaluate:deterministic-v1" ? "Evaluating…" : "Run reference evaluation"}</button></div></>; })()}</article>{providers.map((item) => { const latest = checks[item.provider_id]?.at(-1); const evaluation = evaluations.filter((value) => value.provider_id === item.provider_id).at(-1); return <article key={item.provider_id} className={item.enabled ? "" : "disabled"}><div className="provider-card-head"><span className={`provider-health ${latest?.status ?? "unchecked"}`} /> <div><strong>{item.provider_id}</strong><small>{item.kind.replaceAll("_", " ")} · {item.model}</small></div><b>{item.enabled ? "Enabled" : "Disabled"}</b></div><dl><dt>Endpoint</dt><dd>{item.endpoint}</dd><dt>Credential source</dt><dd>{item.credential_env || "None"}</dd><dt>Configuration</dt><dd>{item.content_hash.slice(0, 16)}…</dd><dt>Last check</dt><dd>{latest ? `${latest.status} · ${(latest.latency_seconds * 1000).toFixed(0)} ms` : "Not checked"}</dd><dt>Evaluation</dt><dd>{evaluation ? `${evaluation.status} · F1 ${evaluation.report.summary.mean_f1.toFixed(3)}` : "Not evaluated"}</dd><dt>Reported tokens</dt><dd>{evaluation?.report.summary.reported_total_tokens?.toLocaleString() ?? "Not reported"}</dd></dl>{latest?.status === "failed" && <p className="provider-check-error">{String(latest.details.error ?? "Provider contract failed")}</p>}<div className="provider-actions"><button disabled={Boolean(busy) || !item.enabled} onClick={() => test(item)}>{busy === `check:${item.provider_id}` ? "Testing…" : "Test handshake"}</button><button disabled={Boolean(busy) || !item.enabled} onClick={() => evaluate(item.provider_id)}>{busy === `evaluate:${item.provider_id}` ? "Evaluating…" : "Run 3-fixture evaluation"}</button><button disabled={Boolean(busy)} onClick={() => toggle(item)}>{item.enabled ? "Disable" : "Enable"}</button></div></article>; })}{!providers.length && <div className="empty-research">No external planners registered. The deterministic offline planner remains available.</div>}</div>
    </div>
    {notice && <div className="agent-notice">{notice}</div>}{error && <div className="form-error">{error}</div>}
  </section><InstallationSettingsPanel /><WorkloadQualificationPanel /><UpdateSettingsPanel /><SecuritySettingsPanel /><BackupSettingsPanel /></>;
}
