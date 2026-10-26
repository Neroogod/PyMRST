import { useEffect, useState } from "react";
import {
  api, type ExampleDefinition, type ProjectRecord, type RunConfiguration,
  type RunEvent, type RunRecord,
} from "../api/client";

const RUNNER_KIND = "pymrst-example";
const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);

export function ExamplesWorkspace({ project }: { project: ProjectRecord }) {
  const [catalog, setCatalog] = useState<ExampleDefinition[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [configurations, setConfigurations] = useState<RunConfiguration[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api.examples(controller.signal)
      .then((items) => { setCatalog(items); setSelectedId((current) => current || items[0]?.example_id || ""); })
      .catch((caught) => { if (caught.name !== "AbortError") setError(caught.message); });
    return () => controller.abort();
  }, []);

  const refreshRuns = async () => {
    const [configValues, runValues] = await Promise.all([
      api.runConfigurations(project.project_id), api.runs(project.project_id),
    ]);
    const exampleConfigurations = configValues.filter((item) => item.runner_kind === RUNNER_KIND);
    const exampleConfigurationIds = new Set(exampleConfigurations.map((item) => item.configuration_id));
    setConfigurations(exampleConfigurations);
    setRuns(runValues.filter((item) => exampleConfigurationIds.has(item.configuration_id)));
  };
  useEffect(() => {
    let active = true;
    refreshRuns().catch((caught) => { if (active) setError(caught instanceof Error ? caught.message : "Could not load runs"); });
    return () => { active = false; };
  }, [project.project_id]);

  useEffect(() => {
    if (!activeRunId) { setEvents([]); return; }
    let active = true;
    const poll = async () => {
      try {
        const [run, eventValues] = await Promise.all([api.run(activeRunId), api.runEvents(activeRunId)]);
        if (!active) return;
        setRuns((current) => current.map((item) => item.run_id === run.run_id ? run : item));
        setEvents(eventValues);
      } catch (caught) { if (active) setError(caught instanceof Error ? caught.message : "Run refresh failed"); }
    };
    poll();
    const timer = window.setInterval(poll, 700);
    return () => { active = false; window.clearInterval(timer); };
  }, [activeRunId]);

  const selected = catalog.find((item) => item.example_id === selectedId);
  const activeRun = runs.find((item) => item.run_id === activeRunId);
  const terminal = activeRun ? TERMINAL.has(activeRun.status) : true;
  const progressEvent = [...events].reverse().find((event) => event.event_type === "progress");
  const resultEvent = [...events].reverse().find((event) => event.event_type === "result");
  const progress = Number(progressEvent?.payload.progress ?? 0);
  const categories = Array.from(new Set(catalog.map((item) => item.category)));

  async function run() {
    if (!selected) return;
    setBusy(true); setError("");
    try {
      const configuration = await api.createRunConfiguration(project.project_id, {
        name: selected.title, runner_kind: RUNNER_KIND, model_id: "",
        payload: { example_id: selected.example_id },
      });
      setConfigurations((current) => [...current, configuration]);
      const runRecord = await api.submitRun(configuration.configuration_id);
      setRuns((current) => [...current, runRecord]);
      setActiveRunId(runRecord.run_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start example");
    } finally { setBusy(false); }
  }
  async function cancel() {
    if (!activeRun) return;
    try { await api.cancelRun(activeRun.run_id); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed"); }
  }

  return <section className="research-room">
    <aside className="study-list">
      <div className="table-heading"><div><span className="eyebrow">Curated library</span><h3>PyMRST examples</h3></div><span>{catalog.length}</span></div>
      {categories.map((category) => <div key={category}>
        <span className="eyebrow">{category}</span>
        {catalog.filter((item) => item.category === category).map((item) => <button
          className={selectedId === item.example_id ? "active" : ""}
          onClick={() => setSelectedId(item.example_id)} key={item.example_id}
        ><strong>{item.title}</strong><span>{item.runtime_class}</span></button>)}
      </div>)}
      {!catalog.length && <div className="empty-research"><strong>{error ? "Could not load catalog" : "Loading catalog…"}</strong></div>}
    </aside>
    <div className="research-main">
      <div className="research-tabs"><span className="eyebrow">Example</span><strong>{selected?.title ?? "Select an example"}</strong></div>
      {selected && <div className="execution-lab">
        <div className="section-lead"><h3>{selected.title}</h3><p>{selected.description}</p></div>
        <p><span className="eyebrow">{selected.category}</span> · runtime class: {selected.runtime_class}. Runs in an isolated process, same as a simulation run — nothing on disk outside its own run directory is touched.</p>
        <button className="button primary" disabled={busy} onClick={run}>{busy ? "Starting…" : "Run example"}</button>
        <div className="execution-list">
          {runs.slice().reverse().map((item) => <button className={activeRunId === item.run_id ? "active" : ""} onClick={() => setActiveRunId(item.run_id)} key={item.run_id}>
            <i className={item.status} />
            <strong>{configurations.find((configuration) => configuration.configuration_id === item.configuration_id)?.name ?? item.run_id}</strong>
            <span>{item.status} · {item.run_id.slice(-7)}</span>
          </button>)}
          {!runs.length && <p>No example runs yet in this project.</p>}
        </div>
        {activeRun && <div className="execution-status">
          <div><span className="eyebrow">Selected run</span><h3>{activeRun.status}</h3><code>{activeRun.run_id}</code></div>
          {!terminal && <button className="button danger" onClick={cancel}>Cancel</button>}
          <div className="live-status">
            <div className={`run-state ${activeRun.status}`}><i />{activeRun.status}</div>
            <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></div>
            <strong>{Math.round(progress * 100)}%</strong>
            <small>{String(progressEvent?.payload.message ?? (terminal ? "Finished" : "Waiting for worker"))}</small>
          </div>
          {resultEvent && <div className="diff-view"><pre>{JSON.stringify(resultEvent.payload.result, null, 2)}</pre></div>}
          <div className="event-log">{events.slice(-12).map((event) => <div key={event.sequence}>
            <code>{String(event.sequence).padStart(4, "0")}</code><span>{event.event_type}</span>
            <strong>{String(event.payload.message ?? event.payload.error_type ?? "")}</strong>
            <small>{event.elapsed_seconds.toFixed(3)}s</small>
          </div>)}</div>
        </div>}
      </div>}
      {error && <div className="form-error research-error">{error}</div>}
    </div>
  </section>;
}
